import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent {
  showCompage = false;
  response:any;
  constructor(
    private http: HttpClient,
  ){
    this.getData();
  }

  getData(){
    this.http.get('assets/response.json').subscribe(res=>{
      console.log(res);
      this.response = res;
     });
  }

  formateResponse(val:any){
    if(val){
      return val.replace(/\n/g, '<br>')
    } else {
      return "";
    }
     
  }
}
